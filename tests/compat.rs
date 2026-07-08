use std::path::PathBuf;

use rsomics_pgen::Pgen;
use rsomics_plink_pca::run_pca;

fn load_golden() -> Pgen {
    let prefix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/test");
    Pgen::load(&prefix).expect("load golden PLINK fileset")
}

#[test]
fn pca_runs_without_error() {
    let pgen = load_golden();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 5, &mut evec, &mut eval).unwrap();
}

#[test]
fn pca_eigenval_count() {
    let pgen = load_golden();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 5, &mut evec, &mut eval).unwrap();
    let eval_str = String::from_utf8(eval).unwrap();
    let n_vals = eval_str.lines().count();
    // Should have min(n_pcs, n_samples - 1) eigenvalues.
    assert!(
        n_vals > 0 && n_vals <= 5,
        "expected 1-5 eigenvalues, got {n_vals}"
    );
}

#[test]
fn pca_eigenvec_has_header() {
    let pgen = load_golden();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 3, &mut evec, &mut eval).unwrap();
    let s = String::from_utf8(evec).unwrap();
    let first = s.lines().next().unwrap();
    assert!(
        first.starts_with("#FID\tIID\tPC1"),
        "unexpected header: {first:?}"
    );
}

#[test]
fn pca_eigenvec_row_count() {
    let pgen = load_golden();
    let n = pgen.n_samples();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 3, &mut evec, &mut eval).unwrap();
    let s = String::from_utf8(evec).unwrap();
    // Header + one row per sample.
    assert_eq!(
        s.lines().count(),
        n + 1,
        "expected {n} data rows + 1 header"
    );
}

#[test]
fn eigenvalues_descending() {
    let pgen = load_golden();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 5, &mut evec, &mut eval).unwrap();
    let vals: Vec<f64> = String::from_utf8(eval)
        .unwrap()
        .lines()
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect();
    for w in vals.windows(2) {
        assert!(w[0] >= w[1] - 1e-10, "eigenvalues not descending: {w:?}");
    }
}

#[test]
fn pca_scores_finite() {
    let pgen = load_golden();
    let mut evec = Vec::new();
    let mut eval = Vec::new();
    run_pca(&pgen, 3, &mut evec, &mut eval).unwrap();
    let s = String::from_utf8(evec).unwrap();
    for line in s.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        for score in &fields[2..] {
            let v: f64 = score.parse().unwrap();
            assert!(v.is_finite(), "non-finite PC score: {score}");
        }
    }
}

#[test]
fn exit_nonzero_on_missing_file() {
    use std::process::Command;
    let bin = env!("CARGO_BIN_EXE_rsomics-plink-pca");
    let status = Command::new(bin)
        .args(["--plink", "/nonexistent/path"])
        .status()
        .expect("spawn binary");
    assert!(!status.success());
}

/// Value-exact regression against `plink2 --pca`, captured from PLINK
/// v2.0.0-a.7.0LM on the committed `test` fileset. plink2 refuses PCA on <50
/// samples without a reference panel, so the golden was produced with the
/// fileset's own `--freq` fed back via `--read-freq` — the same in-sample
/// standardization this crate uses. Eigenvalues are compared numerically;
/// eigenvectors are compared up to a per-column sign flip, which is
/// mathematically arbitrary and even varies between plink2 invocations.
#[test]
fn pca_matches_plink2_golden() {
    use std::process::Command;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let tmp = tempfile::Builder::new()
        .prefix("plink-pca-compat")
        .tempdir_in("/Volumes/KIOXIA/tmp")
        .or_else(|_| tempfile::tempdir())
        .unwrap();
    let out = tmp.path().join("ours");
    let status = Command::new(env!("CARGO_BIN_EXE_rsomics-plink-pca"))
        .args([
            "--plink",
            dir.join("test").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--pcs",
            "9",
        ])
        .status()
        .expect("spawn binary");
    assert!(status.success());

    let read_vals = |p: PathBuf| -> Vec<f64> {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    };
    let gv = read_vals(dir.join("test.eigenval.golden"));
    let ov = read_vals(out.with_extension("eigenval"));
    assert_eq!(gv.len(), ov.len(), "eigenvalue count");
    for (g, o) in gv.iter().zip(&ov) {
        assert!(
            (g - o).abs() <= 1e-4 * g.abs().max(1.0),
            "eigenvalue golden {g} vs ours {o}"
        );
    }

    // eigenvectors: parse to columns [pc][sample], compare up to per-column sign
    let read_vecs = |p: PathBuf| -> Vec<Vec<f64>> {
        let txt = std::fs::read_to_string(p).unwrap();
        let mut rows = Vec::new();
        for line in txt.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            rows.push(
                f[2..]
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect::<Vec<_>>(),
            );
        }
        rows
    };
    let gvec = read_vecs(dir.join("test.eigenvec.golden"));
    let ovec = read_vecs(out.with_extension("eigenvec"));
    assert_eq!(gvec.len(), ovec.len(), "sample count");
    let npc = gvec[0].len();
    for c in 0..npc {
        // sign-align on the column's largest-magnitude golden entry
        let piv = (0..gvec.len())
            .max_by(|&a, &b| gvec[a][c].abs().partial_cmp(&gvec[b][c].abs()).unwrap())
            .unwrap();
        let flip = (gvec[piv][c] >= 0.0) != (ovec[piv][c] >= 0.0);
        for r in 0..gvec.len() {
            let o = if flip { -ovec[r][c] } else { ovec[r][c] };
            assert!(
                (gvec[r][c] - o).abs() <= 1e-4,
                "PC{} sample {}: golden {} vs ours {o}",
                c + 1,
                r + 1,
                gvec[r][c]
            );
        }
    }
}
