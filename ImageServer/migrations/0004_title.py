# Generated by Django 4.1.5 on 2023-02-09 08:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageServer', '0003_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='title',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
            ],
        ),
    ]