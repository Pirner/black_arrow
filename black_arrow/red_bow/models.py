from django.db import models

# Create your models here.


class DataSet(models.Model):
    description = models.TextField()
    name = models.CharField(max_length=512)
