import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):

    _,_,exec_tb = error_detail.exc_info()
    filename = exec_tb.tb_frame.f_code.co_filename
    error_message = "Error message has occured on script [{0}], line number [{1}] error message [{2}]".format(filename, exec_tb.tb_lineno, str(error))
    print("hello")
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        super().__init__(self)
        self.error_massage = error_message_detail(error, error_detail)
        print(self.error_massage)

    def __str__(self):
        return self.error_message

