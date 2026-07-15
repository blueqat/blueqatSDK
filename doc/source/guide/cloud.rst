Cloud access
============

:mod:`blueqat.cloud` provides the groundwork for API-key based access to the
Blueqat cloud service.

API keys
--------

Credentials resolve in this order:

1. ``blueqat.cloud.configure(api_key=...)`` in the current process,
2. the ``BLUEQAT_API_KEY`` environment variable,
3. the config file ``~/.blueqat/config.json``.

.. code-block:: python

   import blueqat.cloud as cloud

   cloud.save_api_key("YOUR_API_KEY")   # persisted with owner-only (0600) permissions
   cloud.get_api_key()                  # resolved key (never logged; masked in repr)
   cloud.delete_api_key()

Submitting circuits
-------------------

Importing :mod:`blueqat.cloud` registers the ``'cloud'`` backend. A submitted
circuit is serialized to the versioned JSON schema together with its run
parameters:

.. code-block:: python

   import blueqat.cloud
   from blueqat import Circuit

   Circuit(2).h[0].cx[0, 1].m[:].run(backend='cloud', shots=100)

Until the public endpoint is live, the default transport raises a clear
error; tests and early integrations can inject their own transport:

.. code-block:: python

   def my_transport(request: dict):
       # request = {"circuit": {...}, "shots": 100, "returns": None, "options": {...}}
       return {"job_id": "...", "status": "queued"}

   blueqat.cloud.configure(api_key="...", transport=my_transport)
