# Print a message when the package is initialized
print("Initializing the 'utilities' package")

# Import specific functions from each module
from .message_template import message_template_1, message_template_2

# Define what will be imported when using `from utilities import *`
__all__ = ["message_template_1", "message_template_2"]
