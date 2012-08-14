try:
    import unittest2 as unittest
except ImportError:
    import unittest
import image_search.db
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

    def test_db(self):
        num_db_images = 1000000
        num_search_images = 3
        num_bytes = 16
        db = image_search.db.LinearHashDB(num_bytes)
        hashes = np.array(np.random.randint(0, 256, (num_db_images, num_bytes)), dtype=np.uint8)
        search_hashes = np.array(np.random.randint(0, 256, (num_search_images, num_bytes)), dtype=np.uint8)
        db.store_hashes(hashes, np.array(np.arange(num_db_images), dtype=np.uint64))
        for result_ids in db.search_hash_nn_multi(search_hashes):
            print(result_ids)
        print(hashes.shape)
        print(search_hashes.shape)

if __name__ == '__main__':
    unittest.main()
