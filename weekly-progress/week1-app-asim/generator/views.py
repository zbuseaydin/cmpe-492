from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Table, Column
from .utils import generate_sql
import json

# Create your views here.

def index(request):
    tables = Table.objects.all()
    return render(request, 'index.html', {'tables': tables})

def add_table(request):
    if request.method == 'POST':
        table_name = request.POST.get('table_name')
        column_names = request.POST.getlist('column_names')
        
        table = Table.objects.create(name=table_name)
        for column_name in column_names:
            Column.objects.create(table=table, name=column_name)
        
        return redirect('index')
    return render(request, 'add_table.html')

def generate_query(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        
        context = "We have a database with tables:\n"
        for table in Table.objects.all():
            columns = ', '.join([col.name for col in table.columns.all()])
            context += f"- {table.name} ({columns})\n"
        
        result = generate_sql(context, question)
        return JsonResponse({'sql_query': result})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

def remove_table(request, table_id):
    if request.method == 'POST':
        table = Table.objects.get(id=table_id)
        table.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'error': 'Invalid request method'}, status=400)
