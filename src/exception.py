import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message including the file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message[{error}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the CustomException with detailed error information.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message
