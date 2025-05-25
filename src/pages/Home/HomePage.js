/* eslint-disable no-use-before-define */
import React, { useState, useEffect } from 'react';
import BookList from '../../components/BookList/BookList';
import BookFeedback from '../../components/BookFeedback/BookFeedback';
import styles from './HomePage.module.css';
import { useUser } from '../../contexts/UserContext'; 
import UserApi from '../../api/UserApi';
import all_books from '../../assets/all_books';

const categories = ["Fiction", "Performing Arts", "Humor", "History", "Self-Help"];

const HomePage = () => {
  const [books, setBooks] = useState([]);
  const [filteredBooks, setFilteredBooks] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [recommendedBooks, setRecommendedBooks] = useState([]);
  const [popularBooks, setPopularBooks] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState();
  const { user } = useUser();
  const defaultRecommendedBooks = [
    {
                "bookId": 84,
                "title": "Mystic River",
                "author": "Brian Helgeland",
                "publisher": "Default Publisher",
                "publicationYear": 2002,
                "isbn": "ISBN586400",
                "genre": "Motion picture plays",
                "descriptions": "Unmarked typescript, dated FINAL DRAFT Rev. 09/16/02 (Pink) Clint Eastwood directed, produced, and scored this mystery drama starring Sean Penn, Tim Robbins, and Kevin Bacon. It as released by Warner Brothers Oct. 3, 2003.",
                "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BMTIzNDUyMjA4MV5BMl5BanBnXkFtZTYwNDc4ODM3._V1_.jpg",
                "quantity": 20,
                "availableQuantity": 2,
                "createdAt": "2024-12-30T15:27:37",
                "previewLink": "https://books.google.com.vn/books?id=WpjzzwEACAAJ&lpg=PP1&dq=intitle:Mystic+River&hl=vi&pg=PP1&output=embed"
            },
            {
                "bookId": 85,
                "title": "Sphere",
                "author": "Michael Crichton",
                "publisher": "Default Publisher",
                "publicationYear": 2011,
                "isbn": "ISBN624731",
                "genre": "Life on other planets",
                "descriptions": "No description available.",
                "coverImageUrl": "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1660273071i/455373.jpg",
                "quantity": 8,
                "availableQuantity": 1,
                "createdAt": "2024-12-30T15:27:37",
                "previewLink": "https://books.google.com.vn/books?id=Ad6CzQEACAAJ&lpg=PP1&dq=intitle:Sphere&hl=vi&pg=PP1&output=embed"
            },
            {
                "bookId": 86,
                "title": "The Pelican Brief",
                "author": "Unknown Author",
                "publisher": "Default Publisher",
                "publicationYear": 2012,
                "isbn": "ISBN831323",
                "genre": "History",
                "descriptions": "No description available.",
                "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BZjA2NmE4MjEtOTkxYy00YjhkLWI2YjgtODFmMGY0Zjc3YTdhXkEyXkFqcGc@._V1_.jpg",
                "quantity": 2,
                "availableQuantity": 16,
                "createdAt": "2024-12-30T15:27:37",
                "previewLink": "https://books.google.com.vn/books?id=-on-zwEACAAJ&lpg=PP1&dq=intitle:The+Pelican+Brief&hl=vi&pg=PP1&output=embed"
            },
  ];

  useEffect(() => {
    const fetchData = async () => {
      await fetchBooks();
      fetchPopularBooks();
      fetchRecommendBooks();
    };
    fetchData();
  }, []);

  useEffect(() => {
    if (books.length > 0 && user?.userId) {
      fetchRecommendBooks();
    }
  }, [books, user]);

  useEffect(() => {
    filterBooksByCategory();
  }, [books, selectedCategory]);

  const fetchBooks = async () => {
    try {
      const response = await UserApi.getBooks();
      if (response.data.code === 0) {
        const fetchBooks = response.data.result.content;
        setBooks(fetchBooks);
        setFilteredBooks(fetchBooks);
      } else {
        console.error("Lỗi: API trả về mã lỗi không phải 0.");
      }
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách:", error);
    }
  };

  const fetchPopularBooks = async () => {
    try {
      const response = await UserApi.getPopularBooks();
      if (response.data.code === 0) {
        const fetchBooks = response.data.result.content;
        setPopularBooks(fetchBooks);
      } else {
        console.error("Lỗi: API trả về mã lỗi không phải 0.");
      }
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách:", error);
    }
  };

 const fetchRecommendBooks = async () => {
  try {
    if (!user || !user.userId) {
      console.error("User ID không hợp lệ");
      return;
    }
    console.log(user.userId);
    const response = await UserApi.getRecommendedBooks(user.userId);

    const recommendedBooksData = response.data.recommendations; // array of book objects

    if (!recommendedBooksData || !Array.isArray(recommendedBooksData)) {
      console.error("Dữ liệu recommendedBooks không đúng định dạng");
      return;
    }

    const recommendedBooks = recommendedBooksData.map((book, index) => ({
      bookId: book.isbn || index,
      title: book.title,
      author: book.author,
      coverImageUrl: book.cover,
      availableQuantity: 5, // giả định, nếu bạn có data thật thì thay thế
    }));

    setRecommendedBooks(recommendedBooks);
    console.log("Danh sách sách đề xuất đã được cập nhật");
  } catch (error) {
    console.error("Lỗi khi lấy danh sách sách:", error);
  }
};


  const filterBooksByCategory = () => {
  if (selectedCategory === "All") {
    setFilteredBooks(all_books);
  } else {
    const filtered = all_books.filter((book) => book.genre === selectedCategory);
    setFilteredBooks(filtered);
  }
};


  return (
    <div className={styles.homePage}>
      <section className={styles.filterSection}>
        <h2>Chọn Thể Loại Sách</h2>
        <div className={styles.categoryList}>
          {[...categories].map((category) => (
            <button
              key={category}
              className={`${styles.categoryButton} ${
                selectedCategory === category ? styles.active : ""
              }`}
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>
      </section>

      <section className={styles.bookSection}>
      <h2>{selectedCategory}</h2> 
      <BookList books={filteredBooks} userId={user?.userId} />
      </section>

      <section className={styles.carouselSection}>
        <h2>Sách Phổ Biến</h2>
        <BookList books={popularBooks} userId={user?.userId} />
      </section>

      <section className={styles.recommendedSection}>
        <h2>Gợi Ý Dành Cho Bạn</h2>
        <BookFeedback books={recommendedBooks.length > 0 ? recommendedBooks : defaultRecommendedBooks} />
      </section>

      {searchResults.length > 0 && (
        <section className={styles.searchResultsSection}>
          <h2>Kết Quả Tìm Kiếm</h2>
          <BookList books={searchResults} />
        </section>
      )}
    </div>
  );
};

export default HomePage;
