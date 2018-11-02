import unittest
import vop
import math

class TestVop(unittest.TestCase):
    def setUp(self):
        pass

    def testSimple(self):
        for s in [ 1,2,3,4,5,6,7,8,9,19,63,64,65,1000,1023,1024,1025, 8191,8292,8193 ]:
            a = vop.arange(s)
            self.assert_(list(a + a) == [x*2 for x in range(s)])

    def testCombo(self):
        s = 10
        a = vop.arange(s)
        self.assert_(list(a+3) == [x+3 for x in range(s)])
        self.assert_(list(3+a) == [3+x for x in range(s)])
        b = a + 15
        self.assert_(list(a*b) == [x*(x+15) for x in range(s)])

    def testSqrt(self):
        s = 1313
        a = vop.sqrt(vop.arange(s))
        for i,x in enumerate(a):
          self.assertAlmostEqual(x, math.sqrt(i), 3)

    def testDiv(self):
        s = 1313
        a = vop.arange(s)
        for i,x in enumerate(7.0 / a):
          if i != 0:
            self.assertAlmostEqual(x, 7.0 / i, 3)
        for i,x in enumerate(a / 7.0):
          self.assertAlmostEqual(x, i / 7.0, 3)
        for i,x in enumerate(a / (a + 10)):
          self.assertAlmostEqual(x, i / float(i + 10), 3)

    def testWhere(self):
        s = 1312
        a = vop.arange(s)
        r = vop.where(a/2 == vop.floor(a/2), 1.0, 0.0)
        self.assert_(sum(r) == s/2)

    def testTake(self):
        data = vop.array('FRED')
        self.assert_(data.tostring() == 'FRED')
        self.assert_(vop.take(data, vop.arange(4)).tostring() == 'FRED')
        self.assert_(vop.take(data, vop.array([3,2,1,0])).tostring() == 'DERF')
        self.assert_(vop.take(data, vop.array([3,2,0,2,1,1,2,3])).tostring() == 'DEFERRED')

    def testTakeB(self):
        data = vop.fromstring('FRED')
        self.assert_(vop.takeB(data, vop.arange(4)).tostring() == 'FRED')
        self.assert_(vop.takeB(data, vop.array([3,2,1,0])).tostring() == 'DERF')
        self.assert_(vop.takeB(data, vop.array([3,2,0,2,1,1,2,3])).tostring() == 'DEFERRED')

    def testAbs(self):
        data = vop.array([-7,7,-1,1])
        self.assert_(sum(data) == 0)
        self.assert_(sum(abs(data)) == 16)

    def testFloor(self):
        data = (vop.arange(32) + 16) * 0.0625
        self.assert_(sum(vop.floor(data)) == 48)

if __name__ == '__main__':
    unittest.main()
