import logging
import configparser
import logging.config
from io import StringIO

def getLogging(confName = "applog"):
    config_path = ".//log//logging.conf"
    
    # 手动读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_string = f.read()
    
    # 使用 StringIO 将字符串转换成文件对象
    from io import StringIO
    config_file = StringIO(config_string)
    
    # 解析配置文件
    cp = configparser.ConfigParser()
    cp.read_file(config_file)
    
    # 将配置应用到 logging
    logging.config.fileConfig(cp)
    
    logger = logging.getLogger(confName)  # 替换为你的 logger 名称
    return logger
    #logging.config.fileConfig(".//log//logging.conf")
#    logging.config.fileConfig(r"C://Users//lenovo//Desktop//AGV//Jsy_project//log//logging.conf",  encoding="utf-8")
    
    #return logging.getLogger(confName)

