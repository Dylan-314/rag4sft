from langchain.docstore.document import Document
from mod.answer_generator.answer_generator import (
    generate_answer,
    make_sample,
    validate_sample,
)

def test_make_and_validate():
    dummy_ctx = [Document(page_content="Paris is the capital of France.")]
    qa = generate_answer("What is the capital of France?", dummy_ctx, model="gpt-4o-mini")
    sample = make_sample("What is the capital of France?", dummy_ctx[0].page_content, qa)
    assert validate_sample(sample)
