from django.test import TestCase


class TestCase(TestCase):
    def setUp(self):
        #Animal.objects.create(name="lion", sound="roar")
        #Animal.objects.create(name="cat", sound="meow")
        pass

    def test_animals_can_speak(self):
        """Animals that can speak are correctly identified"""
        #lion = Animal.objects.get(name="lion")
        #cat = Animal.objects.get(name="cat")
        #self.assertEqual(lion.speak(), 'The lion says "roar"')
        #self.assertEqual(cat.speak(), 'The cat says "meow"')
        self.assertEqual('test', 'test')
        self.assertEqual(1, 2)
