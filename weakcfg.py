#
# read weak.ini
#
# e.g. weakcfg.get("wsprmon", "mycall") -> None or "W1XXX"
#

import ConfigParser

def get(program, key):
    cfg = ConfigParser.SafeConfigParser()
    cfg.read(['weak-local.cfg', 'weak.cfg'])

    if cfg.has_option(program, key):
        return cfg.get(program, key)

    return None
