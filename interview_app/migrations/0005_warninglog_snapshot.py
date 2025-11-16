# Generated migration to add snapshot field to WarningLog
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interview_app', '0004_codesubmission_gemini_evaluation'),
    ]

    operations = [
        migrations.AddField(
            model_name='warninglog',
            name='snapshot',
            field=models.CharField(blank=True, help_text='Filename of the snapshot image captured when warning occurred', max_length=255, null=True),
        ),
    ]

