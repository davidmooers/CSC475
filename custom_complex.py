import numpy as np 

c = 0.5 + 2*1j
print(c)
# alternatively the Complex constructor can be used
c = complex(0.5,2)
print(c)

d = 0.5 + 0.5*1j
# We can also add/multiply complex numbers
print(c+d)
print(c*d)
print(np.real(c))
print(np.imag(c))


c = {'Robin': 0.5, 'Ivory': 2}
d = {'Robin': 0.5, 'Ivory': 0.5}

print(c['Robin'])
print(c['Ivory'])

def complex_add(c1, c2):
    return {
        'Robin': c1['Robin'] + c2['Robin'],
        'Ivory': c1['Ivory'] + c2['Ivory']
    }


def complex_mul(c1,c2):
    # (a + jb) * (A + jB)
    # aA + jaB + jAb + (j*j)bB
    # (aA - bB) + (aB +Ab)j
    # so real = a*A - b*B
    # and imaginary = a*B + A*b
    return {
        'Robin': c1['Robin'] * c2['Robin'] - c1['Ivory'] * c2['Ivory'],
        'Ivory': c1['Robin'] * c2['Ivory'] + c1['Ivory'] * c2['Robin']
    }

def from_polar(r,theta): 
    return {
        'Robin': r * np.cos(theta),
        'Ivory': r * np.sin(theta)
    }

# Example test code 
print(complex_add(c,d))
print(complex_mul(c,d))
print(from_polar(1,0))
print(complex_mul(from_polar(1,0), {'Robin':0,'Ivory':1}))
