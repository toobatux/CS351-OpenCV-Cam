from django.db import models
import datetime

# Create your models here.
class Alerts(models.Model):
    alertDate = models.DateField(auto_now_add=True)
    alertStartTime = models.CharField(max_length=8)
    alertEndTime = models.CharField(max_length=8)