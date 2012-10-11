try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
# Cheat Sheet (method/test) <http://docs.python.org/library/unittest.html>
#
# assertEqual(a, b)       a == b   
# assertNotEqual(a, b)    a != b    
# assertTrue(x)     bool(x) is True  
# assertFalse(x)    bool(x) is False  
# assertRaises(exc, fun, *args, **kwds) fun(*args, **kwds) raises exc
# assertAlmostEqual(a, b)  round(a-b, 7) == 0         
# assertNotAlmostEqual(a, b)          round(a-b, 7) != 0
# 
# Python 2.7+ (or using unittest2)
#
# assertIs(a, b)  a is b
# assertIsNot(a, b) a is not b
# assertIsNone(x)   x is None
# assertIsNotNone(x)  x is not None
# assertIn(a, b)      a in b
# assertNotIn(a, b)   a not in b
# assertIsInstance(a, b)    isinstance(a, b)
# assertNotIsInstance(a, b) not isinstance(a, b)
# assertRaisesRegexp(exc, re, fun, *args, **kwds) fun(*args, **kwds) raises exc and the message matches re
# assertGreater(a, b)       a > b
# assertGreaterEqual(a, b)  a >= b
# assertLess(a, b)      a < b
# assertLessEqual(a, b) a <= b
# assertRegexpMatches(s, re) regex.search(s)
# assertNotRegexpMatches(s, re)  not regex.search(s)
# assertItemsEqual(a, b)    sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)      all the key/value pairs in a exist in b

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        import image_search
        features = np.random.random((100, 10))
        hasher = image_search.MedianHasher().train(features)
        hashes = hasher(features)
        print(hasher)
        print image_search.hash_bit_mean(hashes)

        hasher = image_search.RRMedianHasher(8).train(features)
        print(hasher)
        hashes = hasher(features)
        print(hasher.median_feature)
        print('ProjShape[%s]' % str(hasher.proj.shape))
        print image_search.hash_bit_mean(hashes)

if __name__ == '__main__':
    unittest.main()
