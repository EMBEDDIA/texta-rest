from rest_framework import permissions

from toolkit.core.project.models import Project


"""
    Only superusers can create new projects
    All project users can perform SAFE and UNSAFE_METHODS on project resources.
"""




# Used inside applications to denote access permissions.
class ProjectResourceAllowed(permissions.BasePermission):
    message = 'Insufficient permissions for this resource.'


    def has_permission(self, request, view):
        return self._permission_check(request, view)


    def has_object_permission(self, request, view, obj):
        return self._permission_check(request, view)


    def _permission_check(self, request, view):
        # retrieve project object
        try:
            project_object = Project.objects.get(id=view.kwargs['project_pk'])
        except:
            return False

        # check if user is listed among project users
        if request.user in project_object.users.all():
            return True

        if request.user in project_object.administrators.all():
            return True

        # check if user is superuser
        if request.user.is_superuser:
            return True
        # nah, not gonna see anything!
        return False


# Used inside the Project endpoints.
class ProjectAllowed(permissions.BasePermission):
    message = 'Insufficient permissions for this project.'


    def has_object_permission(self, request, view, obj):
        return self._permission_check(request, view)


    def _permission_check(self, request, view):
        # always permit SAFE_METHODS and superuser
        if request.user.is_superuser:
            return True
        # retrieve project object
        try:
            project_object = Project.objects.get(id=view.kwargs['pk'])
        except:
            return False

        # Project admins have the right to edit project information.
        if request.user in project_object.administrators.all() and request.method in permissions.SAFE_METHODS:
            return True

        # Project users are permitted safe access to project list_view
        if request.user in project_object.users.all() and request.method in permissions.SAFE_METHODS:
            return True
        return False


class IsSuperUser(permissions.BasePermission):

    def has_permission(self, request, view):
        return self._permission_check(request, view)


    def has_object_permission(self, request, view, obj):
        return self._permission_check(request, view)


    def _permission_check(self, request, view):
        return request.user and request.user.is_superuser


class ExtraActionResource(ProjectResourceAllowed):
    """ Overrides ProjectResourceAllowed for project extra_actions that use POST """


    def _permission_check(self, request, view):
        # retrieve project object
        try:
            project_object = Project.objects.get(id=view.kwargs['pk'])
        except:
            return False
        # check if user is listed among project users
        if request.user in project_object.users.all():
            return True
        # check if user is superuser
        if request.user.is_superuser:
            return True
        # nah, not goa see anything!
        return False


class UserIsAdminOrReadOnly(permissions.BasePermission):
    ''' custom class for user_profile '''


    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        else:
            return request.user.is_superuser


    def has_object_permission(self, request, view, obj):
        # can't edit original admin
        if obj.pk == 1 and request.method not in permissions.SAFE_METHODS:
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        else:
            return request.user.is_superuser
