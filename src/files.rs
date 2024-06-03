use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter},
    marker::PhantomData,
    path::Path,
};

use anyhow::{anyhow, Context};
use serde::{de::DeserializeOwned, Serialize};

type Result<T> = anyhow::Result<T>;

pub fn read_json_file_if_exists<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match fs::File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            Ok(serde_json::from_reader(reader)?)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).context(format!("reading main data file '{:?}'", path)),
    }
}

pub fn replace_json_file<T: Serialize>(path: &Path, data: &T) -> Result<()> {
    let temp_file_path = {
        if let Some(file_name) = path.file_name() {
            let mut new_file_name = file_name.to_os_string();
            new_file_name.push(".new");
            path.with_file_name(new_file_name)
        } else {
            return Err(anyhow!("Bad file path: {:?}", path));
        }
    };

    {
        let f = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&temp_file_path)?;
        let mut writer = BufWriter::new(f);
        serde_json::to_writer(&mut writer, data)?;
    }

    fs::rename(&temp_file_path, &path)
        .with_context(|| format!("failed to rename '{:?}' -> '{:?}'", temp_file_path, path))?;
    Ok(())
}

pub struct JsonLinesReader<R: BufRead, T: DeserializeOwned> {
    reader: R,
    buf: String,
    _phantom: PhantomData<T>,
}

pub fn read_json_lines_from_file_if_exists<T: DeserializeOwned>(
    path: &Path,
) -> Result<JsonLinesReader<BufReader<File>, T>> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => File::open("/dev/null")?,
        Err(e) => return Err(e.into()),
    };
    let r = BufReader::new(f);
    Ok(read_json_lines_from_reader(r))
}

pub fn read_json_lines_from_reader<R: BufRead, T: DeserializeOwned>(
    reader: R,
) -> JsonLinesReader<R, T> {
    JsonLinesReader::<R, T> {
        reader,
        buf: String::new(),
        _phantom: PhantomData,
    }
}

impl<R: BufRead, T: DeserializeOwned> Iterator for JsonLinesReader<R, T> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.buf.clear();
        match self.reader.read_line(&mut self.buf) {
            Ok(0) => None,
            Ok(_) => Some(serde_json::from_str(&self.buf).map_err(|e| e.into())),
            Err(e) => Some(Err(e.into())),
        }
    }
}
