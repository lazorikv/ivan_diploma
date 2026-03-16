from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Look up a key in a dict: {{ my_dict|get_item:key }}"""
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None
