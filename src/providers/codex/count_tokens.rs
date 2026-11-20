use super::translate::request::{
    ResponsesContentPart, ResponsesFunctionCallOutput, ResponsesFunctionCallOutputContentPart,
    ResponsesInputItem, ResponsesRequest, ResponsesTool,
};
use tiktoken_rs::o200k_base_singleton;

/// Approximate token counter for Codex translated requests.
/// Text uses the OpenAI `o200k_base` tokenizer. Images, encrypted reasoning,
/// and protocol framing still use local estimates, so the result is not a
/// provider billing count.
pub fn count_translated_tokens(translated: &ResponsesRequest) -> u64 {
    count_translated_tokens_with(translated, &approx_token_count)
}

/// Tokenizer-free estimate used to seed `message_start.usage.input_tokens`.
/// The terminal `message_delta` carries the provider-reported usage, so this
/// only needs to be close enough for the live progress display.
pub fn estimate_translated_tokens(translated: &ResponsesRequest) -> u64 {
    count_translated_tokens_with(translated, &approx_token_count_from_bytes)
}

fn approx_token_count_from_bytes(text: &str) -> u64 {
    u64::try_from(text.len()).unwrap_or(u64::MAX).div_ceil(4)
}

fn count_translated_tokens_with(
    translated: &ResponsesRequest,
    text_tokens: &dyn Fn(&str) -> u64,
) -> u64 {
    let mut total = 0u64;

    // Instructions
    if let Some(ref instructions) = translated.instructions {
        total += text_tokens(instructions);
    }

    // Input items
    for item in &translated.input {
        total += count_input_item_tokens(item, text_tokens);
    }

    // Tools
    if let Some(ref tools) = translated.tools {
        total += count_tool_tokens(tools, text_tokens);
    }

    // Overhead
    total += translated.input.len() as u64 * 4;
    total += translated.tools.as_ref().map_or(0, |t| t.len() as u64 * 4);

    // Model name
    total += text_tokens(&translated.model);

    total.max(1)
}

fn count_input_item_tokens(item: &ResponsesInputItem, text_tokens: &dyn Fn(&str) -> u64) -> u64 {
    match item {
        ResponsesInputItem::AdditionalTools { tools, .. } => tools
            .iter()
            .map(|tool| text_tokens(&serde_json::to_string(tool).unwrap_or_default()))
            .sum(),
        ResponsesInputItem::Message { content, .. } => {
            let mut total = 0u64;
            for part in content {
                total += count_content_part_tokens(part, text_tokens);
            }
            total
        }
        ResponsesInputItem::FunctionCall {
            name, arguments, ..
        } => text_tokens(name) + text_tokens(arguments),
        ResponsesInputItem::FunctionCallOutput { output, .. } => {
            count_function_call_output_tokens(output, text_tokens)
        }
        ResponsesInputItem::Reasoning {
            encrypted_content, ..
        }
        | ResponsesInputItem::Compaction { encrypted_content } => {
            approx_reasoning_token_count(encrypted_content)
        }
        ResponsesInputItem::CompactionTrigger => 0,
    }
}

fn count_function_call_output_tokens(
    output: &ResponsesFunctionCallOutput,
    text_tokens: &dyn Fn(&str) -> u64,
) -> u64 {
    match output {
        ResponsesFunctionCallOutput::Text(text) => text_tokens(text),
        ResponsesFunctionCallOutput::ContentItems(content) => content
            .iter()
            .map(|part| match part {
                ResponsesFunctionCallOutputContentPart::InputText { text } => text_tokens(text),
                ResponsesFunctionCallOutputContentPart::InputImage { .. } => 2000,
            })
            .sum(),
    }
}

fn approx_reasoning_token_count(encoded_content: &str) -> u64 {
    let model_visible_bytes = encoded_content
        .len()
        .saturating_mul(3)
        .checked_div(4)
        .unwrap_or(0)
        .saturating_sub(650);
    u64::try_from(model_visible_bytes.saturating_add(3) / 4).unwrap_or(u64::MAX)
}

fn count_content_part_tokens(
    part: &ResponsesContentPart,
    text_tokens: &dyn Fn(&str) -> u64,
) -> u64 {
    match part {
        ResponsesContentPart::InputText { text } => text_tokens(text),
        ResponsesContentPart::OutputText { text } => text_tokens(text),
        ResponsesContentPart::InputImage { .. } => 2000, // Image token estimate
    }
}

fn count_tool_tokens(tools: &[ResponsesTool], text_tokens: &dyn Fn(&str) -> u64) -> u64 {
    let mut total = 0u64;
    for tool in tools {
        match tool {
            ResponsesTool::Function(f) => {
                total += text_tokens(&f.name);
                if let Some(ref desc) = f.description {
                    total += text_tokens(desc);
                }
                total += text_tokens(&serde_json::to_string(&f.parameters).unwrap_or_default());
            }
            ResponsesTool::WebSearch(_) => {
                total += 10; // fixed overhead for web search tool
            }
        }
    }
    total
}

pub(crate) fn approx_token_count(text: &str) -> u64 {
    u64::try_from(o200k_base_singleton().count_ordinary(text)).unwrap_or(u64::MAX)
}

pub(crate) fn truncate_to_token_budget(text: &str, max_tokens: u64) -> String {
    let max_tokens = usize::try_from(max_tokens).unwrap_or(usize::MAX);
    if max_tokens == 0 {
        return String::new();
    }

    let tokenizer = o200k_base_singleton();
    let tokens = tokenizer.encode_ordinary(text);
    if tokens.len() <= max_tokens {
        return text.to_string();
    }

    // A token boundary may split a multi-byte UTF-8 scalar. Back up until the
    // decoded prefix is valid rather than replacing bytes or splitting text.
    for end in (1..=max_tokens).rev() {
        if let Ok(prefix) = tokenizer.decode(&tokens[..end]) {
            return prefix;
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn count_simple_request() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let count = count_translated_tokens(&req);
        assert!(count > 0);
    }

    #[test]
    fn count_request_with_tools() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "use tool"}]}],
            "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let count = count_translated_tokens(&req);
        assert!(count > 0);
    }

    #[test]
    fn count_is_monotonic() {
        let short: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        let long: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "this is a much longer message with many words in it"}]}],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        assert!(count_translated_tokens(&long) >= count_translated_tokens(&short));
    }

    #[test]
    fn estimate_tracks_the_accurate_count_without_a_tokenizer() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "instructions": "You are a helpful assistant. ".repeat(100),
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "The quick brown fox jumps over the lazy dog. ".repeat(500)
                }]
            }],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();

        let accurate = count_translated_tokens(&req);
        let estimate = estimate_translated_tokens(&req);

        assert!(estimate > 0);
        assert!(
            estimate * 2 > accurate && estimate < accurate * 2,
            "estimate {estimate} should stay within 2x of accurate {accurate}"
        );
    }

    #[test]
    fn estimate_keeps_empty_requests_nonzero() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.5",
            "input": [],
            "store": false,
            "stream": true,
            "parallel_tool_calls": true,
            "text": {"verbosity": "low"},
        }))
        .unwrap();
        assert!(estimate_translated_tokens(&req) > 0);
    }

    #[test]
    fn o200k_counts_unbroken_text_instead_of_collapsing_it() {
        assert_eq!(approx_token_count(&"a".repeat(4096)), 512);
        assert_eq!(approx_token_count(&"汉字测试上下文压缩".repeat(384)), 2688);
        assert_eq!(
            approx_token_count(&"QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo".repeat(112)),
            2240
        );
        assert_eq!(
            approx_token_count(&"function f(a){return a===0?1:a*f(a-1)};".repeat(96)),
            1632
        );
    }

    #[test]
    fn truncation_respects_o200k_budget_and_utf8_boundaries() {
        for text in [
            "a".repeat(4096),
            "汉字测试上下文压缩".repeat(384),
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo".repeat(112),
        ] {
            let truncated = truncate_to_token_budget(&text, 100);
            assert!(text.starts_with(&truncated));
            assert!(approx_token_count(&truncated) <= 100);
            assert!(!truncated.is_empty());
        }
    }

    #[test]
    fn structured_tool_output_counts_text_and_images() {
        let text = ResponsesFunctionCallOutput::Text("caption".to_string());
        let mixed = ResponsesFunctionCallOutput::ContentItems(vec![
            ResponsesFunctionCallOutputContentPart::InputText {
                text: "caption".to_string(),
            },
            ResponsesFunctionCallOutputContentPart::InputImage {
                image_url: "data:image/png;base64,YQ==".to_string(),
                detail: None,
            },
        ]);

        assert_eq!(
            count_function_call_output_tokens(&mixed, &approx_token_count),
            count_function_call_output_tokens(&text, &approx_token_count) + 2000
        );
    }

    #[test]
    fn encrypted_reasoning_uses_codex_model_visible_size_estimate() {
        let encoded_content = "A".repeat(4000);
        assert_eq!(approx_reasoning_token_count(&encoded_content), 588);
        assert_eq!(approx_reasoning_token_count("short"), 0);
    }
}
