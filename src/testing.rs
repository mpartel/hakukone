use std::{fs, path::PathBuf};

// https://stackoverflow.com/a/40234666/965979
#[macro_export]
macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let mut name = type_name_of(f);
        name = name.strip_suffix("::f").unwrap();
        while let Some(prefix) = name.strip_suffix("::{{closure}}") {
            name = prefix;
        }
        name
    }};
}

pub fn clear_and_get_test_dir(test_name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("./target/tests/{test_name}"));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("failed to clear test directory");
    }
    dir
}
