import vivilux as vl
import vivilux.photonics

dummyLayer = vl.Layer(4)
mzi = vl.photonics.MZImesh(4, dummyLayer)

mat = mzi.matrix

print("Unitary?")
print((mat @ mat.T.conj()).round(2))