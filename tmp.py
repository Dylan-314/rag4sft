from pathlib import Path

# 要处理的目标文件名
TARGET_FILES = [
    "streamlit_app.py",
    "chat_service.py",
    "answer_generator.py",
    "generator.py",
    "manager.py",
]

INSERT_BLOCK = [
    "import sys, os",
    "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \"../../\")))",
]

def patch_file(file: Path):
    lines = file.read_text(encoding="utf-8").splitlines()
    # 如果已添加，则跳过
    if any("sys.path.append" in line for line in lines[:5]):
        print(f"✅ 已处理: {file}")
        return

    new_lines = INSERT_BLOCK + [""] + lines
    file.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"🛠️ 插入成功: {file}")

def run_patch(root="mod"):
    for file in Path(root).rglob("*.py"):
        if file.name in TARGET_FILES:
            patch_file(file)

if __name__ == "__main__":
    run_patch()