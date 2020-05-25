""" Logger for printing """

# Settings
interval = 50


# Logger
loggers = {}
step = 0

def add(name, val):
    global loggers, step
    loggers[name] = val

def tick():
    global loggers, step
    if step > 0 and step % interval == 0:
        # print(f'\n $$$ Step {step}')
        print('')
        for name in sorted(loggers):
            val = loggers[name]
            if val is not None:
                print('$ ',str(name).ljust(15), ':', val)
    step += 1

def reset():
    global loggers, step
    loggers = {}

import torch
