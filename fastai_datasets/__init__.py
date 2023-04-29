__version__ = "0.0.4"

# Bugfix: fastprogress bars not showing in VSCode notebooks. Taken from https://github.com/microsoft/vscode-jupyter/issues/13163
import IPython
def update_patch(self, obj):
    IPython.display.clear_output(wait=True)
    self.display(obj)
IPython.display.DisplayHandle.update = update_patch