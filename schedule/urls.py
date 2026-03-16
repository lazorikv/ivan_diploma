from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("generate/", views.generate, name="generate"),
    path("schedule/", views.schedule_view, name="schedule"),
    path("data/", views.data_view, name="data"),
    path("export/", views.export_view, name="export"),
    path("upload/", views.upload_files, name="upload_files"),
    path("upload/delete/<str:file_type>/", views.delete_upload, name="delete_upload"),
]
