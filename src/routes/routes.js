import HomePage from '../pages/Home/HomePage.js';
import DefaultLayout from '../layouts/userLayout'; 
import BookDetails from '../pages/BookDetails/BookDetails';
import BorrowBookPage from '../pages/BorrowBookPage/BorrowBookPage';
import BorrowManagementPage from '../pages/BorrowManagementPage/BorrowManagementPage';

import Login from '../pages/Login/Login.js'
const publicRoutes = [
  {path: '/', component: HomePage, layout: DefaultLayout},
  {path: '/login', component: Login, layout: DefaultLayout},
  {path: '/books', component: BookDetails, layout: DefaultLayout, childPath:':bookId'},
  {path: '/borrow', component: BorrowBookPage, layout: DefaultLayout, childPath:':bookId'},
  {path: '/borrowBook', component: BorrowManagementPage, layout: DefaultLayout},




];

export { publicRoutes };
