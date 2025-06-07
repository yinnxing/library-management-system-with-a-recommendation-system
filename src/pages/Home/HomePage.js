/* eslint-disable no-use-before-define */
import React, { useState, useEffect } from 'react';
import BookList from '../../components/BookList/BookList';
import '../../styles/design-system.css';
import styles from './HomePage.module.css';
import { useUser } from '../../contexts/UserContext'; 
import UserApi from '../../api/UserApi';

const categories = ["All", "Fiction", "Biography & Autobiography", "Humor", "History", "Self-Help", "Body, Mind & Spirit", "True Crime", "Adultery", "Drama"];

const getCategoryIcon = (category) => {
  const iconMap = {
    "All": "📚",
    "Fiction": "📖",
    "Biography & Autobiography": "👤",
    "Humor": "😄",
    "History": "🏛️",
    "Self-Help": "💪",
    "Body, Mind & Spirit": "🧘",
    "True Crime": "🔍",
    "Adultery": "💔",
    "Drama": "🎭",
    "Family & Relationships": "👨‍👩‍👧‍👦"
  };
  return iconMap[category] || "📚";
};

const HomePage = () => {
  const [books, setBooks] = useState([]);
  const [popularBooks, setPopularBooks] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);
  const [loading, setLoading] = useState(false);
  const { user } = useUser();

  const booksPerPage = 12;

  useEffect(() => {
    fetchPopularBooks();
  }, []);

  useEffect(() => {
    fetchBooksByCategory();
  }, [selectedCategory, currentPage]);

  const fetchBooksByCategory = async () => {
    setLoading(true);
    try {
      const response = await UserApi.getBooksByCategory(
        currentPage, 
        booksPerPage, 
        selectedCategory === "All" ? null : selectedCategory
      );
      
      if (response.data.code === 0) {
        const result = response.data.result;
        setBooks(result.content);
        setTotalPages(result.totalPages);
        setTotalElements(result.totalElements);
      } else {
        console.error("Lỗi: API trả về mã lỗi không phải 0.");
        setBooks([]);
      }
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách:", error);
      setBooks([]);
    } finally {
      setLoading(false);
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
      console.error("Lỗi khi lấy danh sách sách phổ biến:", error);
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setCurrentPage(1); // Reset to first page when category changes
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
    // Scroll to top when page changes
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    const pages = [];
    const maxVisiblePages = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage + 1 < maxVisiblePages) {
      startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }

    // Previous button
    if (currentPage > 1) {
      pages.push(
        <button
          key="prev"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage - 1)}
        >
          ‹ Trước
        </button>
      );
    }

    // First page
    if (startPage > 1) {
      pages.push(
        <button
          key={1}
          className={styles.paginationButton}
          onClick={() => handlePageChange(1)}
        >
          1
        </button>
      );
      if (startPage > 2) {
        pages.push(
          <span key="ellipsis1" className={styles.paginationEllipsis}>
            ...
          </span>
        );
      }
    }

    // Page numbers
    for (let i = startPage; i <= endPage; i++) {
      pages.push(
        <button
          key={i}
          className={`${styles.paginationButton} ${
            currentPage === i ? styles.active : ''
          }`}
          onClick={() => handlePageChange(i)}
        >
          {i}
        </button>
      );
    }

    // Last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        pages.push(
          <span key="ellipsis2" className={styles.paginationEllipsis}>
            ...
          </span>
        );
      }
      pages.push(
        <button
          key={totalPages}
          className={styles.paginationButton}
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </button>
      );
    }

    // Next button
    if (currentPage < totalPages) {
      pages.push(
        <button
          key="next"
          className={styles.paginationButton}
          onClick={() => handlePageChange(currentPage + 1)}
        >
          Tiếp ›
        </button>
      );
    }

    return pages;
  };

  return (
    <div className={styles.homePage}>
      {/* Category Filter Section */}
      <section className={styles.filterSection}>
        <div className={styles.filterHeader}>
          <h2 className={styles.filterTitle}>
            <span className={styles.filterIcon}>📚</span>
            Khám Phá Thể Loại Sách
          </h2>
          <p className={styles.filterSubtitle}>
            Chọn thể loại để tìm những cuốn sách phù hợp với sở thích của bạn
          </p>
        </div>
        <div className={styles.categoryList}>
          {categories.map((category) => (
            <button
              key={category}
              className={`${styles.categoryButton} ${
                selectedCategory === category ? styles.active : ""
              }`}
              onClick={() => handleCategoryChange(category)}
            >
              <span className={styles.categoryIcon}>
                {getCategoryIcon(category)}
              </span>
              {category === "All" ? "Tất cả" : category}
            </button>
          ))}
        </div>
      </section>

      {/* Books Section */}
      <section className={styles.bookSection}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>
            {selectedCategory === "All" ? "Tất cả sách" : selectedCategory}
          </h2>
          <div className={styles.resultsInfo}>
            {loading ? (
              <span className={styles.loadingText}>Đang tải...</span>
            ) : (
              <span className={styles.resultsCount}>
                Hiển thị {books.length} trong tổng số {totalElements} cuốn sách
              </span>
            )}
          </div>
        </div>

        {loading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.loadingSpinner}></div>
            <p className={styles.loadingMessage}>Đang tải danh sách sách...</p>
          </div>
        ) : books.length > 0 ? (
          <>
            <BookList books={books} userId={user?.userId} />
            
            {/* Pagination */}
            {totalPages > 1 && (
              <div className={styles.paginationContainer}>
                <div className={styles.paginationInfo}>
                  Trang {currentPage} / {totalPages}
                </div>
                <div className={styles.pagination}>
                  {renderPagination()}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📚</div>
            <h3 className={styles.emptyTitle}>Không tìm thấy sách nào</h3>
            <p className={styles.emptyDescription}>
              Hiện tại không có sách nào trong thể loại "{selectedCategory}". 
              Hãy thử chọn thể loại khác.
            </p>
          </div>
        )}
      </section>

      {/* Popular Books Section */}
      <section className={styles.carouselSection}>
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>
            <span className={styles.sectionIcon}>🔥</span>
            Sách Phổ Biến
          </h2>
          <p className={styles.sectionSubtitle}>
            Những cuốn sách được yêu thích nhất tại thư viện
          </p>
        </div>
        <BookList books={popularBooks} userId={user?.userId} />
      </section>
    </div>
  );
};

export default HomePage;
