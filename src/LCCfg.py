__author__ = 'Rays'

import ConfigParser


class LCCfg:
    DEFAULT_SECTION = "DEFAULT"

    def __init__(self, fn):
        config = ConfigParser.RawConfigParser()
        config.optionxform = str  #reference: http://docs.python.org/library/configparser.html
        config.read(fn)
        self.appdb_url = config.get(self.DEFAULT_SECTION, "app_db.url")
        self.appdb_port = int(config.get(self.DEFAULT_SECTION, "app_db.port"))
        self.appdb_user = config.get(self.DEFAULT_SECTION, "app_db.user")
        self.appdb_password = config.get(self.DEFAULT_SECTION, "app_db.password")
        self.appdb_instance = config.get(self.DEFAULT_SECTION, "app_db.instance")
        self.data_dir = config.get(self.DEFAULT_SECTION, "data.dir")
        self.data_fileprefix = config.get(self.DEFAULT_SECTION, "data.fileprefix")
        self.data_fileextension = config.get(self.DEFAULT_SECTION, "data.fileextension")
        self.dictionary_file = config.get(self.DEFAULT_SECTION, "dictionary.file")
        self.dictionary_dictsheet = config.get(self.DEFAULT_SECTION, "dictionary.dictsheet")
        self.sampling_file = config.get(self.DEFAULT_SECTION, "sampling.file")
        self.sql_dir = config.get(self.DEFAULT_SECTION, "sql.dir")
        self.sql_fileextension = config.get(self.DEFAULT_SECTION, "sql.fileextension")

config = LCCfg("default.cfg")
print ("jdbc:mysql:%s:%s/%s  -  %s/%s " %
       (config.appdb_url, str(config.appdb_port), config.appdb_instance, config.appdb_user, config.appdb_password))
