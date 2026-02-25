import React, { useState, useEffect, useCallback } from 'react';
import { initializeApp } from 'firebase/app';
import { 
  getAuth, 
  onAuthStateChanged, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signInWithPopup, 
  GoogleAuthProvider, 
  signOut,
  updateProfile
} from 'firebase/auth';
import { 
  Sun, Moon, MapPin, Mail, Lock, LogOut, Loader2, 
  BarChart3, AlertCircle, Cloud, CloudRain, Sun as SunIcon, 
  TrendingUp, Activity, CheckCircle2, User
} from 'lucide-react';

// ==========================================
// 1. FIREBASE CONFIGURATION
// ==========================================
// In a production environment, these would be in your .env file.
// The canvas provides __firebase_config dynamically if connected.
const getFirebaseConfig = () => {
  try {
    if (typeof __firebase_config !== 'undefined' && __firebase_config) {
      return JSON.parse(__firebase_config);
    }
  } catch (e) {
    console.warn("Could not parse dynamic Firebase config, using fallbacks.", e);
  }
  return {
    apiKey: "DEMO_KEY",
    authDomain: "demo-app.firebaseapp.com",
    projectId: "demo-app",
  };
};

const app = initializeApp(getFirebaseConfig());
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

// ==========================================
// 2. CUSTOM HOOKS
// ==========================================

// Hook for Real-Time Clock & Date
const useCurrentTime = () => {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  return time;
};

// Hook for Geolocation & Weather (Open-Meteo & BigDataCloud)
const useWeather = () => {
  const [weather, setWeather] = useState({ 
    temp: null, 
    condition: null, 
    city: 'Detecting location...',
    loading: true 
  });

  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const { latitude, longitude } = position.coords;
            
            // 1. Get City Name (Free Reverse Geocoding)
            const geoRes = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${latitude}&longitude=${longitude}&localityLanguage=en`);
            const geoData = await geoRes.json();
            const city = geoData.city || geoData.locality || 'Unknown City';

            // 2. Get Weather (Free Open-Meteo API)
            const weatherRes = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`);
            const weatherData = await weatherRes.json();
            
            setWeather({
              temp: Math.round(weatherData.current_weather.temperature),
              condition: weatherData.current_weather.weathercode, // Weather code logic can be expanded
              city: city,
              loading: false
            });
          } catch (e) {
            setWeather({ temp: null, condition: null, city: 'Weather unavailable', loading: false });
          }
        },
        () => {
          setWeather({ temp: null, condition: null, city: 'Location access denied', loading: false });
        }
      );
    } else {
      setWeather({ temp: null, condition: null, city: 'Geolocation unsupported', loading: false });
    }
  }, []);

  return weather;
};

// Hook for Dark/Light Theme
const useTheme = () => {
  const [theme, setTheme] = useState('dark'); // Default to dark for modern data apps

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  
  return { theme, toggleTheme };
};

// ==========================================
// 3. UI COMPONENTS
// ==========================================

const GoogleIcon = () => (
  <svg viewBox="0 0 24 24" className="w-5 h-5 mr-2" xmlns="http://www.w3.org/2000/svg">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg>
);

const ThemeToggle = ({ theme, toggleTheme }) => (
  <button
    onClick={toggleTheme}
    className="absolute top-4 right-4 p-2 rounded-full bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-700 transition-colors z-50"
    aria-label="Toggle theme"
  >
    {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
  </button>
);

const WeatherWidget = () => {
  const { temp, city, loading, condition } = useWeather();
  const time = useCurrentTime();

  const formattedTime = time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const formattedDate = time.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });

  return (
    <div className="mt-auto pt-10 text-slate-100">
      <div className="flex flex-col gap-6 backdrop-blur-md bg-white/10 p-6 rounded-2xl border border-white/20 shadow-xl">
        {/* Clock & Date */}
        <div>
          <h2 className="text-4xl font-bold tracking-tight mb-1">{formattedTime}</h2>
          <p className="text-slate-300 font-medium">{formattedDate}</p>
        </div>
        
        <div className="h-px w-full bg-white/20" />
        
        {/* Weather */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <MapPin className="w-5 h-5 text-blue-400" />
            <span className="font-medium text-lg">{loading ? 'Locating...' : city}</span>
          </div>
          {!loading && temp !== null && (
            <div className="flex items-center gap-2">
              <span className="text-2xl font-semibold">{temp}°C</span>
              {/* Very basic condition mapping - customize as needed */}
              {temp > 20 ? <SunIcon className="w-6 h-6 text-yellow-400" /> : <Cloud className="w-6 h-6 text-slate-300" />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 4. MAIN PAGES
// ==========================================

const Dashboard = ({ user }) => {
  const handleLogout = () => signOut(auth);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 p-8 transition-colors duration-300 flex flex-col items-center">
      <div className="w-full max-w-5xl">
        <header className="flex justify-between items-center mb-12 bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold">Nexus Forecast</h1>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center text-blue-700 dark:text-blue-300 font-bold border border-blue-200 dark:border-blue-800">
                {user?.email?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="hidden md:block">
                <p className="text-sm font-medium">{user?.displayName || 'Forecaster'}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">{user?.email}</p>
              </div>
            </div>
            <button 
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>
        </header>

        <main className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 bg-white dark:bg-slate-800 p-8 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 min-h-[400px] flex flex-col items-center justify-center text-center">
             <Activity className="w-16 h-16 text-blue-500 mb-4 opacity-50" />
             <h2 className="text-2xl font-bold mb-2">Workspace Ready</h2>
             <p className="text-slate-500 dark:text-slate-400 max-w-md">
               Authentication successful. This is your placeholder dashboard where the Time Series Forecasting tools and charts will be integrated.
             </p>
          </div>
          <div className="flex flex-col gap-6">
             <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500" /> System Status
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Firebase Auth Connected
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Location API Active
                  </div>
                  <div className="flex items-center gap-3 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Ready for ML Backend
                  </div>
                </div>
             </div>
          </div>
        </main>
      </div>
    </div>
  );
};

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  // Helper for human-readable Firebase errors
  const parseError = (err) => {
    const code = err.code || '';
    if (code === 'auth/invalid-credential' || code === 'auth/user-not-found' || code === 'auth/wrong-password') return 'Invalid email or password.';
    if (code === 'auth/email-already-in-use') return 'An account with this email already exists.';
    if (code === 'auth/weak-password') return 'Password should be at least 6 characters.';
    if (code === 'auth/invalid-email') return 'Please enter a valid email address.';
    if (code === 'auth/popup-closed-by-user') return 'Google sign-in was cancelled.';
    return String(err.message || 'An unexpected error occurred.');
  };

  const handleEmailAuth = async (e) => {
    e.preventDefault();
    if (!email || !password) return setError('Please fill in all required fields.');
    if (!isLogin && !name) return setError('Please provide your name.');

    setLoading(true);
    setError('');
    setSuccessMsg('');

    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        // Set display name for new user
        await updateProfile(userCredential.user, { displayName: name });
      }
    } catch (err) {
      setError(parseError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleAuth = async () => {
    setLoading(true);
    setError('');
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err) {
      setError(parseError(err));
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setSuccessMsg('');
    setEmail('');
    setPassword('');
    setName('');
  };

  return (
    <div className="flex min-h-screen w-full bg-slate-50 dark:bg-slate-900 transition-colors duration-300">
      
      {/* LEFT PANEL: Branding & Widgets (Hidden on small screens) */}
      <div className="hidden lg:flex flex-1 relative flex-col justify-between p-12 overflow-hidden">
        {/* Dynamic Abstract Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900 via-indigo-900 to-slate-900 dark:from-slate-900 dark:via-blue-950 dark:to-indigo-950">
           {/* Abstract Data Shapes */}
           <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-500/20 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/3" />
           <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-indigo-500/20 rounded-full blur-[120px] translate-y-1/3 -translate-x-1/4" />
        </div>
        
        {/* Branding Content */}
        <div className="relative z-10 text-white">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-blue-500/20 p-3 rounded-xl border border-blue-400/30 backdrop-blur-sm">
              <BarChart3 className="w-8 h-8 text-blue-300" />
            </div>
            <h1 className="text-3xl font-bold tracking-wide">Nexus<span className="font-light">Forecast</span></h1>
          </div>
          <p className="text-xl text-blue-100/80 max-w-md font-light leading-relaxed">
            Enterprise-grade time series forecasting. Authenticate to access intelligent predictive modeling and live data analytics.
          </p>
        </div>

        {/* Real-time Widget */}
        <div className="relative z-10">
          <WeatherWidget />
        </div>
      </div>

      {/* RIGHT PANEL: Authentication Form */}
      <div className="w-full lg:w-[540px] flex items-center justify-center p-8 bg-white dark:bg-slate-900 relative shadow-2xl z-20 transition-colors duration-300 border-l border-transparent dark:border-slate-800">
        <div className="w-full max-w-md space-y-8 relative">
          
          <div className="text-center lg:text-left space-y-2">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
              {isLogin ? 'Welcome back' : 'Create an account'}
            </h2>
            <p className="text-slate-500 dark:text-slate-400">
              {isLogin 
                ? 'Enter your credentials to access your dashboard.' 
                : 'Sign up to start forecasting your time series data.'}
            </p>
          </div>

          {/* Alert Messages */}
          {error && (
            <div className="flex items-center gap-2 p-4 text-sm text-red-600 bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 rounded-xl">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <p>{error}</p>
            </div>
          )}
          {successMsg && (
            <div className="flex items-center gap-2 p-4 text-sm text-green-600 bg-green-50 dark:bg-green-500/10 border border-green-200 dark:border-green-500/20 rounded-xl">
              <CheckCircle2 className="w-5 h-5 shrink-0" />
              <p>{successMsg}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleEmailAuth} className="space-y-5">
            {!isLogin && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Full Name</label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-slate-400" />
                  </div>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="block w-full pl-10 pr-3 py-2.5 border border-slate-300 dark:border-slate-700 rounded-xl leading-5 bg-transparent dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-all"
                    placeholder="John Doe"
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Email Address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-slate-400" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-300 dark:border-slate-700 rounded-xl leading-5 bg-transparent dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-all"
                  placeholder="name@company.com"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Password</label>
                {isLogin && (
                  <a href="#" className="text-sm font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400">
                    Forgot password?
                  </a>
                )}
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-slate-400" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-300 dark:border-slate-700 rounded-xl leading-5 bg-transparent dark:bg-slate-800 text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-all"
                  placeholder="••••••••"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-xl shadow-sm text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 focus:ring-offset-slate-50 dark:focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all mt-6"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          <div className="mt-8">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-300 dark:border-slate-700" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-3 bg-white dark:bg-slate-900 text-slate-500 transition-colors">
                  Or continue with
                </span>
              </div>
            </div>

            <div className="mt-6">
              <button
                onClick={handleGoogleAuth}
                disabled={loading}
                className="w-full flex items-center justify-center py-3 px-4 border border-slate-300 dark:border-slate-700 rounded-xl shadow-sm bg-white dark:bg-slate-800 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 focus:ring-offset-slate-50 dark:focus:ring-offset-slate-900 transition-all disabled:opacity-50"
              >
                <GoogleIcon />
                Google
              </button>
            </div>
          </div>

          <p className="mt-8 text-center text-sm text-slate-600 dark:text-slate-400">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button
              onClick={toggleMode}
              className="font-semibold text-blue-600 hover:text-blue-500 dark:text-blue-400 transition-colors"
            >
              {isLogin ? 'Sign up' : 'Sign in'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 5. ROOT APP COMPONENT
// ==========================================

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const [user, setUser] = useState(null);
  const [loadingUser, setLoadingUser] = useState(true);

  // Authentication State Observer
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoadingUser(false);
    });
    return () => unsubscribe();
  }, []);

  if (loadingUser) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 flex items-center justify-center transition-colors">
        <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
      </div>
    );
  }

  return (
    <>
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
      {user ? <Dashboard user={user} /> : <AuthPage />}
    </>
  );
}