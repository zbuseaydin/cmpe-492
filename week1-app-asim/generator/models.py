from django.db import models

# Create your models here.
class Table(models.Model):
    name = models.CharField(max_length=100)

class Column(models.Model):
    table = models.ForeignKey(Table, on_delete=models.CASCADE, related_name='columns')
    name = models.CharField(max_length=100)