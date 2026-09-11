import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'state/navigation_state_provider.dart';
import 'ui/map_screen.dart';
import 'ui/theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
  ));

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(
            create: (_) => NavigationStateProvider()..initialise()),
      ],
      child: const NavPulseApp(),
    ),
  );
}

class NavPulseApp extends StatelessWidget {
  const NavPulseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NavPulse',
      debugShowCheckedModeBanner: false,
      theme: NavTheme.dark(),
      home: const MapScreen(),
    );
  }
}
