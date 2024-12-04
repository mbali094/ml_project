import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):

    _,_,exec_tb = error_detail.exc_info()
    filename = exec_tb.tb_frame.f_code.co_filename
    error_message = "Error message has occured on script [{0}], line number [{1}] error message [{2}]".format(filename, exec_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_massage = error_message_detail(error_message, error_detail = error_detail)

    def __str__(str):
        return self.error_message