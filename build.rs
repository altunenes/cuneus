fn main() {
    // static linking of bzip2
    println!("cargo:rustc-link-lib=static=bz2");
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/bzip2/lib");
}