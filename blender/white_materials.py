"""White-material proxy setup kept isolated from the Python runtime."""

WHITE_MATERIAL_NAME = "ProxyWhiteMaterial"
WHITE_MATERIAL_COLOR = (0.8, 0.8, 0.8, 1.0)


def apply_white_materials(bpy):
    material = bpy.data.materials.get(WHITE_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(WHITE_MATERIAL_NAME)
    material.diffuse_color = WHITE_MATERIAL_COLOR
    for obj in bpy.context.scene.objects:
        if hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(material)
    return material
