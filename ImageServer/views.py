from django.shortcuts import render
from ImageServer.models import Image

# Create your views here.

def main(request):
    images = Image.objects.all()
    print(images)
    args = {'images': images}
    return render(request, 'main.html', args) #value übergibt den Parameter and die HTML-Seite und kann dort als "image" und "label" verwendet werden