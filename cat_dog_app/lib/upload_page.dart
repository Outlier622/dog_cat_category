import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class UploadPage extends StatefulWidget {
  @override
  _UploadPageState createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  File? _image;

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });

      await _classifyImage(_image!);
    }
  }

  Future<void> _classifyImage(File imageFile) async {
    final uri = Uri.parse('http://192.168.1.110:5000/classify');
    final request = http.MultipartRequest('POST', uri);
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    try {
      final response = await request.send();
      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final result = jsonDecode(responseBody);

        _showResultDialog(result);
      } else {
        _showErrorDialog('Classification failed with status ${response.statusCode}');
      }
    } catch (e) {
      _showErrorDialog('Error occurred: $e');
    }
  }

  void _showResultDialog(Map<String, dynamic> result) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Classification Result'),
        content: Text(_formatResult(result)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }

  String _formatResult(Map<String, dynamic> result) {
    return result.entries.map((e) => '${e.key}: ${e.value}').join('\n');
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Upload Photo')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: Icon(Icons.camera_alt),
              label: Text('Capture Image'),
            ),
            SizedBox(height: 10),
            _image != null
                ? Image.file(_image!, height: 200)
                : Container(height: 200, color: Colors.grey[300]),
          ],
        ),
      ),
    );
  }
}
