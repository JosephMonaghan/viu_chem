import pymzml

path = "/Users/josephmonaghan/Dropbox/AAAA - Post Doc/viu_chem/viu_chem/tests/Datafiles/Demo_01.mzML"

my_file = pymzml.run.Reader(path)

# test = my_file.get_spectrum_count()
# spec = my_file[test]
# print(spec)

spec = my_file[10]

print(spec['MS:1000511'])

# if spec["MS:1000129"]:
#     print("scan is negative mode!")
# elif spec["MS:1000130"]:
#     print("scan is positive mode!")



# for key in spec.keys():
#     print(key)

# print(len(my_file))

# print(len(my_file))