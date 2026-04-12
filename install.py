import launch

DEPENDENCIES = ["requests"]

for package in DEPENDENCIES:
    if not launch.is_installed(package):
        launch.run_pip(f"install {package}", f"civitai_helper: {package}")
