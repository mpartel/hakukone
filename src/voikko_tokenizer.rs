use std::{mem, rc::Rc};

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use voikko_rs::voikko::{TokenType, Voikko};

type Result<T> = anyhow::Result<T>;

// Alternatively we could check if Voikko is thread-safe and modify voikko_rs to impl Sync.
thread_local!(static VOIKKO: Result<Rc<VoikkoTokenizer>> = {
    let tokenizer = VoikkoTokenizer::new()?;
    Ok(Rc::new(tokenizer))
});

pub struct VoikkoTokenizer {
    voikko: Voikko,
}

impl VoikkoTokenizer {
    pub fn new() -> Result<VoikkoTokenizer> {
        Ok(VoikkoTokenizer {
            voikko: Voikko::new("fi", None).context("initializing libvoikko (is it installed?)")?,
        })
    }

    pub fn thread_local() -> Result<Rc<VoikkoTokenizer>> {
        VOIKKO.with(|t| match t.as_ref() {
            Ok(v) => Ok(v.clone()),
            Err(e) => Err(anyhow!("Voikko init error: {}", e)),
        })
    }

    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut byte_offset = 0u64;
        let mut tokens = Vec::<Token>::new();
        for token in self.voikko.tokens(text) {
            byte_offset += token.token_text.as_bytes().len() as u64;
            if token.token_type == TokenType::Word {
                let analyses = self.voikko.analyze(&token.token_text);
                let mut lemmas = Vec::<String>::with_capacity(analyses.len());
                for mut analysis in analyses.into_iter() {
                    // TODO: use WORDBASES?
                    // TODO: filter out some tokens, e.g. CLASS: lyhenne?
                    if let Some(baseform_in_map) = analysis.get_mut("BASEFORM") {
                        let mut baseform = String::new();
                        mem::swap(baseform_in_map, &mut baseform);
                        lemmas.push(baseform);
                    }
                }
                tokens.push(Token {
                    lemmas,
                    byte_offset,
                })
            }
        }
        tokens
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Token {
    #[serde(rename = "l")]
    pub lemmas: Vec<String>,
    #[serde(rename = "o")]
    pub byte_offset: u64,
}
