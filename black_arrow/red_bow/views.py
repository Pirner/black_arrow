from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from skimage.filters import threshold_sauvola

import datetime


def current_datetime(request):
    #now = datetime.datetime.now()
    #html = "<html><body>It is now %s.</body></html>" % now
    #return HttpResponse(html)
    template = loader.get_template('red_bow/base.html')
    context = {
        'data': 3,
    }
    return HttpResponse(template.render(context, request))
