# Generated by Django 4.1.5 on 2023-01-29 18:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('image_id', models.AutoField(primary_key=True, serialize=False)),
                ('image', models.ImageField(upload_to='images/')),
            ],
        ),
    ]