import os

def __autoimport(path):
    import os
    from os.path import join as opj
    is_module = lambda x: x.endswith('.py') and not x.startswith('__init__')

    ret = []
    for dirpath, dirnames, filenames in os.walk(path):
        modules = [os.path.splitext(f)[0] for f in filter(is_module, filenames)]
        relpath = os.path.relpath(dirpath, path).split(os.sep)
        for module in modules:
            imp = module
            if relpath != ['.'] :
                imp = '.'.join(relpath + [module])
            tmp = __import__(imp, globals=globals(), fromlist=['*'], level=1)
            if hasattr(tmp, '__all__'):
                ret += tmp.__all__
                for name in tmp.__all__:
                    globals()[name] = vars(tmp)[name]
    return ret


__all__ = __autoimport(__path__[0])

if not 'PLAMSDEFAULTS' in os.environ :
        os.environ['PLAMSDEFAULTS'] = os.path.join(__path__[0],'plams_defaults')
