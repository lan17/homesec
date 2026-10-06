//! Refuse accidental system/shared OpenCV selection when Cargo is invoked directly.

use std::{env, fs, path::PathBuf};

const BUILD_HINT: &str = "Use make rust-build or native/webrtc/build_native.py to build the pinned static native libraries";

fn required_environment(key: &str) -> String {
    println!("cargo:rerun-if-env-changed={key}");
    env::var(key).unwrap_or_else(|_| panic!("Missing {key}. {BUILD_HINT}"))
}

fn require_environment(key: &str, expected: &str) {
    assert_eq!(
        required_environment(key),
        expected,
        "Unexpected {key}. {BUILD_HINT}"
    );
}

fn main() {
    let prefix = PathBuf::from(required_environment("HOMESEC_OPENCV_PREFIX"));
    assert!(
        prefix.is_absolute(),
        "OpenCV prefix must be absolute. {BUILD_HINT}"
    );
    let version = required_environment("HOMESEC_OPENCV_VERSION");
    let include = prefix.join("include/opencv4");
    let library = prefix.join("lib");
    let third_party = library.join("opencv4/3rdparty");
    require_environment("OPENCV_INCLUDE_PATHS", &include.to_string_lossy());
    require_environment(
        "OPENCV_LINK_PATHS",
        &format!("{},{}", library.display(), third_party.display()),
    );
    let system_libraries = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "c++",
        Ok("linux") => "stdc++,m,dl,pthread,rt",
        _ => panic!("Bundled OpenCV supports Linux and macOS. {BUILD_HINT}"),
    };
    require_environment(
        "OPENCV_LINK_LIBS",
        &format!("static=opencv_imgproc,static=opencv_core,static=zlib,{system_libraries}"),
    );
    require_environment(
        "OPENCV_DISABLE_PROBES",
        "pkg_config,cmake,vcpkg_cmake,vcpkg",
    );
    let header = include.join("opencv2/core/version.hpp");
    for path in [
        prefix.join(".complete"),
        header.clone(),
        library.join("libopencv_imgproc.a"),
        library.join("libopencv_core.a"),
        third_party.join("libzlib.a"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
        assert!(
            path.is_file(),
            "Missing private native archive/header. {BUILD_HINT}"
        );
    }
    let definitions = fs::read_to_string(header).expect("OpenCV version header is readable");
    let header_version: Vec<&str> = [
        "CV_VERSION_MAJOR",
        "CV_VERSION_MINOR",
        "CV_VERSION_REVISION",
    ]
    .into_iter()
    .map(|name| {
        definitions
            .lines()
            .find_map(|line| {
                let mut words = line.split_whitespace();
                if words.next() == Some("#define") && words.next() == Some(name) {
                    words.next()
                } else {
                    None
                }
            })
            .expect("OpenCV version header defines its numeric version")
    })
    .collect();
    assert_eq!(
        header_version.join("."),
        version,
        "OpenCV header differs from the native pin"
    );
}
