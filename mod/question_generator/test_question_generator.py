from langchain.docstore.document import Document
from mod.question_generator.generator import generate_questions

def test_generate_questions_format():
    dummy_chunk = Document(page_content="Python is a programming language.")
    qa = generate_questions([dummy_chunk], n_per_chunk=1, lang="en", model="gpt-4o-mini")  # 便宜模型即可
    assert qa and isinstance(qa[0], dict)
    assert "chunk_id" in qa[0] and "question" in qa[0]