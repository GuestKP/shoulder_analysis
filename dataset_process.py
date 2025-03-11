import c3d

filename = 'datasets\\AMASS\\ACCAD\\Male1Walking\\Walk B1 - Stand to Walk.c3d'

reader = c3d.Reader(open(filename, 'rb'))

for i, points, analog in reader.read_frames():
    print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))