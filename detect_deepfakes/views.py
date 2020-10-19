from django.shortcuts import render
from django.http import HttpResponse
from .index import classify_image

def upload_img(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        result = classify_image(uploaded_file)
        if result == 'Fake':
            return HttpResponse("You are fake")
        elif result == 'Real':
            return HttpResponse("You are real!")
        else:
            return HttpResponse(result)
    return render(request, 'index.html')


