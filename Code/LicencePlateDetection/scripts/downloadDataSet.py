from roboflow import Roboflow

rf = Roboflow(api_key="bsnNjnp2rjMXtruRvCLU")
project = rf.workspace("max-mustermann-gmm7j").project("german-license-plates-hptbz")
version = project.version(7)
dataset = version.download("yolov8")