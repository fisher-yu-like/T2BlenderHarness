from pathlib import Path
import tomllib


def test_project_declares_video_runtime_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with (project_root / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    dependencies = set(pyproject["project"]["dependencies"])

    assert {
        "pydantic>=2.0,<3.0",
        "typing-extensions>=4.12",
        "Pillow>=10.0",
        "imageio>=2.34",
        "imageio-ffmpeg>=0.5",
    } <= dependencies
