import HomePage from '../pages/Home/HomePage.js';
import DefaultLayout from '../layouts/userLayout'; 
import BookDetails from '../pages/BookDetails/BookDetails';
import BorrowBookPage from '../pages/BorrowBookPage/BorrowBookPage';
import BorrowManagementPage from '../pages/BorrowManagementPage/BorrowManagementPage';
import AdminDashboard from '../pages/admin/AdminDashboard';
import Login from '../pages/Login/Login.js'
import Signup from '../pages/Signup/Signup.js'
import BookManagement from '../pages/admin/bookmanagement/BookManagement.js';
import FavoriteBookPage from '../pages/FavoritePage/FavoriteBookPage.js';
import TransactionManagement from '../pages/admin/TransactionManagement.js';
import UserManagement from '../pages/admin/UserManagement.js';
import UserProfile from '../pages/UserProfile/UserProfile.js';
import OAuthCallback from '../pages/OAuthCallback/OAuthCallback.js';
import AdvancedSearchPage from '../pages/AdvancedSearchPage/AdvancedSearchPage.tsx';

const publicRoutes = [
  {path: '/', component: HomePage, layout: DefaultLayout},
  {path: '/login', component: Login, layout: DefaultLayout},
  {path: '/signup', component: Signup, layout: DefaultLayout},
  {path: '/books', component: BookDetails, layout: DefaultLayout, childPath:':bookId'},
  {path: '/borrow', component: BorrowBookPage, layout: DefaultLayout, childPath:':bookId'},
  {path: '/favorite', component: FavoriteBookPage, layout: DefaultLayout},
  {path: '/borrowBook', component: BorrowManagementPage, layout: DefaultLayout},
  {path: '/user-profile', component: UserProfile, layout: DefaultLayout},
  {path: '/oauth2/callback', component: OAuthCallback, layout: DefaultLayout},
  {path: '/advanced-search', component: AdvancedSearchPage, layout: DefaultLayout},
];


const adminRoutes = [
  {
    path: '/admin',
    component: AdminDashboard, 
    children: [
      { path: 'books', component: BookManagement },
      { path: 'users', component: UserManagement },
      { path: 'transactions', component: TransactionManagement },
    ]
  },
];

export { publicRoutes, adminRoutes };
