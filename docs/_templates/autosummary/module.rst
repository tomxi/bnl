{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% if functions %}

.. rubric:: Functions

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if classes %}

.. rubric:: Classes

.. autosummary::
   :toctree:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}

.. rubric:: Exceptions

.. autosummary::
   :toctree:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}

.. rubric:: Attributes

.. autosummary::
   :toctree:
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if modules %}

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %} 